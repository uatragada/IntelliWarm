import argparse
import json

from intelliwarm.learning.policy_catalog import build_policy_catalog, evaluate_named_policies


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic IntelliWarm policies across scenario-library rollouts."
    )
    parser.add_argument(
        "--policy",
        dest="policies",
        action="append",
        help="Named policy to evaluate. Repeat to compare multiple policies.",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help="Scenario name to evaluate. Repeat to limit comparison to selected scenarios.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on rollout steps per scenario.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable summary.",
    )
    return parser


def _summary_payload(results):
    return {
        policy_name: {
            "scenario_count": summary.scenario_count,
            "total_reward": summary.total_reward,
            "total_cost": summary.total_cost,
            "total_comfort_violation": summary.total_comfort_violation,
            "scenario_results": [
                {
                    "scenario_name": result.scenario_name,
                    "steps": result.steps,
                    "total_reward": result.total_reward,
                    "total_cost": result.total_cost,
                    "total_comfort_violation": result.total_comfort_violation,
                    "final_zone_heat_sources": result.final_zone_heat_sources,
                }
                for result in summary.scenario_results
            ],
        }
        for policy_name, summary in results.items()
    }


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    selected_policies = args.policies or ["eco-electric", "comfort-electric", "comfort-furnace"]
    results = evaluate_named_policies(
        selected_policies,
        scenario_names=args.scenarios,
        max_steps=args.max_steps,
    )
    payload = _summary_payload(results)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for policy_name, summary in payload.items():
        print(
            f"{policy_name}: reward={summary['total_reward']:.3f}, "
            f"cost={summary['total_cost']:.3f}, "
            f"comfort_violation={summary['total_comfort_violation']:.3f}, "
            f"scenarios={summary['scenario_count']}"
        )
        for result in summary["scenario_results"]:
            print(
                f"  - {result['scenario_name']}: steps={result['steps']}, "
                f"reward={result['total_reward']:.3f}, "
                f"cost={result['total_cost']:.3f}, "
                f"comfort_violation={result['total_comfort_violation']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
