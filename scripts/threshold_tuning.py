#!/usr/bin/env python3
"""
Fase 3 — Face Verification Threshold Tuning Tool

Usage:
    # 1. Enroll test subjects with multiple photos
    python scripts/threshold_tuning.py enroll --subject USR-1001 --images ./samples/person_a/

    # 2. Run genuine tests (same person, should match)
    python scripts/threshold_tuning.py genuine --subject USR-1001 --images ./samples/person_a_test/

    # 3. Run impostor tests (different person, should NOT match)
    python scripts/threshold_tuning.py impostor --subject USR-1001 --images ./samples/person_b/

    # 4. Analyze collected scores and recommend thresholds
    python scripts/threshold_tuning.py analyze

    # 5. Quick test a single image against a subject
    python scripts/threshold_tuning.py test --subject USR-1001 --image ./test.jpg

Environment:
    FACE_SERVICE_URL  (default: http://localhost:8001)
    FACE_API_KEY      (default: change-me-to-a-long-random-string)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import httpx

# ── Configuration ──────────────────────────────────────────────
BASE_URL = os.getenv("FACE_SERVICE_URL", "http://localhost:8001")
API_KEY = os.getenv("FACE_API_KEY", "change-me-to-a-long-random-string")
RESULTS_DIR = Path(__file__).parent / "tuning_results"
RESULTS_FILE = RESULTS_DIR / "scores.json"

HEADERS = {"X-API-Key": API_KEY}
TIMEOUT = 30.0


def load_scores() -> dict:
    """Load existing score data."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"genuine": [], "impostor": [], "metadata": {}}


def save_scores(data: dict):
    """Persist score data."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(data, indent=2))


def get_image_files(path: str) -> list[Path]:
    """Get all image files from a directory or single file."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    return []


# ── Commands ───────────────────────────────────────────────────

def cmd_enroll(args):
    """Enroll face templates for a subject."""
    images = get_image_files(args.images)
    if not images:
        print(f"❌ No images found in: {args.images}")
        return

    print(f"\n📸 Enrolling {len(images)} image(s) for subject: {args.subject}")
    print(f"   Service: {BASE_URL}\n")

    for img_path in images:
        with open(img_path, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/v1/faces/enroll",
                headers=HEADERS,
                files={"image": (img_path.name, f, "image/jpeg")},
                data={"subject_id": args.subject},
                timeout=TIMEOUT,
            )

        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ {img_path.name} → quality: {data.get('quality_score', 'N/A')}")
        else:
            print(f"   ❌ {img_path.name} → {resp.status_code}: {resp.text}")


def cmd_genuine(args):
    """Test genuine pairs (same person — should match)."""
    images = get_image_files(args.images)
    if not images:
        print(f"❌ No images found in: {args.images}")
        return

    scores_data = load_scores()
    print(f"\n🔬 Genuine test: {len(images)} image(s) vs subject {args.subject}\n")

    for img_path in images:
        with open(img_path, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/v1/faces/verify",
                headers=HEADERS,
                files={"image": (img_path.name, f, "image/jpeg")},
                data={"subject_id": args.subject},
                timeout=TIMEOUT,
            )

        if resp.status_code == 200:
            data = resp.json()
            score = data.get("score", 0)
            match = data.get("match", False)
            icon = "✅" if match else "⚠️"
            print(f"   {icon} {img_path.name} → score: {score:.4f} (match: {match})")
            scores_data["genuine"].append({
                "subject": args.subject,
                "image": img_path.name,
                "score": score,
                "match": match,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            print(f"   ❌ {img_path.name} → {resp.status_code}: {resp.text}")

    save_scores(scores_data)
    print(f"\n   Saved {len(scores_data['genuine'])} genuine scores to {RESULTS_FILE}")


def cmd_impostor(args):
    """Test impostor pairs (different person — should NOT match)."""
    images = get_image_files(args.images)
    if not images:
        print(f"❌ No images found in: {args.images}")
        return

    scores_data = load_scores()
    print(f"\n🔬 Impostor test: {len(images)} image(s) vs subject {args.subject}\n")

    for img_path in images:
        with open(img_path, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/v1/faces/verify",
                headers=HEADERS,
                files={"image": (img_path.name, f, "image/jpeg")},
                data={"subject_id": args.subject},
                timeout=TIMEOUT,
            )

        if resp.status_code == 200:
            data = resp.json()
            score = data.get("score", 0)
            match = data.get("match", False)
            icon = "❌" if match else "✅"  # Impostor matching = bad
            print(f"   {icon} {img_path.name} → score: {score:.4f} (match: {match})")
            scores_data["impostor"].append({
                "subject": args.subject,
                "image": img_path.name,
                "score": score,
                "match": match,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            print(f"   ❌ {img_path.name} → {resp.status_code}: {resp.text}")

    save_scores(scores_data)
    print(f"\n   Saved {len(scores_data['impostor'])} impostor scores to {RESULTS_FILE}")


def cmd_analyze(args):
    """Analyze collected scores and recommend thresholds."""
    scores_data = load_scores()
    genuine = [s["score"] for s in scores_data.get("genuine", [])]
    impostor = [s["score"] for s in scores_data.get("impostor", [])]

    if not genuine and not impostor:
        print("❌ No scores collected yet. Run 'genuine' and 'impostor' tests first.")
        return

    print("\n" + "=" * 60)
    print("📊 THRESHOLD ANALYSIS REPORT")
    print("=" * 60)

    if genuine:
        g_min = min(genuine)
        g_max = max(genuine)
        g_avg = sum(genuine) / len(genuine)
        print(f"\n🟢 Genuine scores ({len(genuine)} samples):")
        print(f"   Min:  {g_min:.4f}")
        print(f"   Max:  {g_max:.4f}")
        print(f"   Avg:  {g_avg:.4f}")

    if impostor:
        i_min = min(impostor)
        i_max = max(impostor)
        i_avg = sum(impostor) / len(impostor)
        print(f"\n🔴 Impostor scores ({len(impostor)} samples):")
        print(f"   Min:  {i_min:.4f}")
        print(f"   Max:  {i_max:.4f}")
        print(f"   Avg:  {i_avg:.4f}")

    if genuine and impostor:
        # Recommend thresholds
        gap_exists = g_min > i_max

        print(f"\n{'─' * 60}")
        print("🎯 RECOMMENDED THRESHOLDS")
        print(f"{'─' * 60}")

        if gap_exists:
            # Clean separation — ideal case
            optimal = (g_min + i_max) / 2
            print(f"\n   ✅ Clean separation detected!")
            print(f"   Lowest genuine:  {g_min:.4f}")
            print(f"   Highest impostor: {i_max:.4f}")
            print(f"\n   Recommended threshold: {optimal:.4f}")
            print(f"\n   ACCEPT zone:  score >= {round(optimal + 0.02, 2)}")
            print(f"   REJECT zone:  score <  {round(optimal - 0.02, 2)}")
            print(f"   REVIEW zone:  {round(optimal - 0.02, 2)} <= score < {round(optimal + 0.02, 2)}")
        else:
            # Overlap — need to balance FAR vs FRR
            overlap_start = max(i_min, g_min)
            overlap_end = min(i_max, g_max)
            middle = (overlap_start + overlap_end) / 2

            print(f"\n   ⚠️  Score overlap detected!")
            print(f"   Overlap range: [{overlap_start:.4f}, {overlap_end:.4f}]")
            print(f"\n   Conservative (low FAR):  {overlap_end:.4f}")
            print(f"   Balanced:               {middle:.4f}")
            print(f"   Permissive (low FRR):   {overlap_start:.4f}")
            print(f"\n   Suggestion: Start with {round(middle, 2)} and adjust based on user feedback.")

        # FAR / FRR at various thresholds
        print(f"\n{'─' * 60}")
        print("📈 FAR / FRR AT VARIOUS THRESHOLDS")
        print(f"{'─' * 60}")
        print(f"   {'Threshold':>10} │ {'FAR':>8} │ {'FRR':>8} │ Notes")
        print(f"   {'─' * 10}─┼{'─' * 9}─┼{'─' * 9}─┼{'─' * 15}")

        for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            far = sum(1 for s in impostor if s >= thr) / len(impostor) * 100
            frr = sum(1 for s in genuine if s < thr) / len(genuine) * 100
            note = ""
            if far == 0 and frr == 0:
                note = "← Perfect"
            elif far == 0:
                note = "← No false accepts"
            elif frr == 0:
                note = "← No false rejects"
            print(f"   {thr:>10.2f} │ {far:>7.1f}% │ {frr:>7.1f}% │ {note}")

    print(f"\n{'=' * 60}")

    # Save recommendation
    scores_data["metadata"]["last_analysis"] = datetime.now().isoformat()
    scores_data["metadata"]["genuine_count"] = len(genuine)
    scores_data["metadata"]["impostor_count"] = len(impostor)
    save_scores(scores_data)


def cmd_test(args):
    """Quick test: verify a single image against a subject."""
    print(f"\n🔍 Testing {args.image} vs subject {args.subject}\n")

    with open(args.image, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/v1/faces/verify",
            headers=HEADERS,
            files={"image": (Path(args.image).name, f, "image/jpeg")},
            data={"subject_id": args.subject},
            timeout=TIMEOUT,
        )

    if resp.status_code == 200:
        data = resp.json()
        icon = "✅ MATCH" if data.get("match") else "❌ NO MATCH"
        print(f"   Result:    {icon}")
        print(f"   Score:     {data.get('score', 'N/A')}")
        print(f"   Threshold: {data.get('threshold', 'N/A')}")
    else:
        print(f"   Error: {resp.status_code} — {resp.text}")


def cmd_status(args):
    """Check enrollment status for a subject."""
    resp = httpx.get(
        f"{BASE_URL}/v1/faces/subjects/{args.subject}",
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n📋 Subject: {data['subject_id']}")
        print(f"   Status:            {data['status']}")
        print(f"   Total embeddings:  {data['total_embeddings']}")
        print(f"   Active embeddings: {data['active_embeddings']}")
    else:
        print(f"   ❌ {resp.status_code}: {resp.text}")


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face Verification Threshold Tuning Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    p = sub.add_parser("enroll", help="Enroll face templates")
    p.add_argument("--subject", required=True, help="Subject ID (e.g. USR-1001)")
    p.add_argument("--images", required=True, help="Path to image or directory")

    # genuine
    p = sub.add_parser("genuine", help="Test genuine pairs (same person)")
    p.add_argument("--subject", required=True)
    p.add_argument("--images", required=True)

    # impostor
    p = sub.add_parser("impostor", help="Test impostor pairs (different person)")
    p.add_argument("--subject", required=True)
    p.add_argument("--images", required=True)

    # analyze
    sub.add_parser("analyze", help="Analyze scores and recommend thresholds")

    # test
    p = sub.add_parser("test", help="Quick single-image test")
    p.add_argument("--subject", required=True)
    p.add_argument("--image", required=True)

    # status
    p = sub.add_parser("status", help="Check subject enrollment status")
    p.add_argument("--subject", required=True)

    args = parser.parse_args()

    commands = {
        "enroll": cmd_enroll,
        "genuine": cmd_genuine,
        "impostor": cmd_impostor,
        "analyze": cmd_analyze,
        "test": cmd_test,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
