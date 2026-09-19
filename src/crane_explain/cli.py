from __future__ import annotations

import argparse
import json

from .io import load_episode
from .realize import render_template
from .reasoning import plan_contrast, plan_recovery_count
from .validation import validate_episode
from .verification import verify_final_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence-checked robot explanations")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("episode")
    explain = sub.add_parser("explain")
    explain.add_argument("episode")
    explain.add_argument("--alternative")
    explain.add_argument("--recovery-count", action="store_true")
    args = parser.parse_args()
    episode = load_episode(args.episode)
    errors = validate_episode(episode)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 2
    if args.command == "validate":
        print(json.dumps({"valid": True, "episode_id": episode.episode_id}, indent=2))
        return 0
    plan = plan_recovery_count(episode) if args.recovery_count else plan_contrast(
        episode, args.alternative)
    text = render_template(plan)
    result = verify_final_text(plan, text)
    print(json.dumps({"plan": plan.to_dict(), "text": text,
                      "verification": result.__dict__}, indent=2))
    return 0 if result.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())

