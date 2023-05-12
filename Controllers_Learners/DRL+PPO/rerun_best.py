from rerunner import AgentRerunner
import modular_robots
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
async def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "body",
        type=str,
        help="The body of the robot.",
    )
    parser.add_argument(
        "num",
        type=str,
        help="The number of the experiment",
    )
    args = parser.parse_args()
    body = args.body
    num = args.num

    file_path = "./data/PPO/"+body+"/database"+num

    rerunner = AgentRerunner()
    body = modular_robots.get(body)
    await rerunner.rerun(body, file_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
