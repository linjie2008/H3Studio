"""End-to-end check: turning the upscale off while the run waits for confirmation
must release the run instead of blocking forever.

Runs a throwaway 2-second t2v shot in a temporary project with
confirm_upscale=True, waits for the "awaiting_upscale" state, turns the
project's SeedVR2 switch off, and verifies the run leaves the waiting state on
its own and finishes with the upscale skipped.
"""
import json
import sys
from pathlib import Path
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner
import store


def main():
    project = store.new_project("临时验证-超分自动跳过")
    project_id = project["id"]
    project.update({"width": 480, "height": 864, "generation_speed": "turbo4",
                    "attention_mode": "sage", "upscale": True, "confirm_upscale": True,
                    "upscale_resolution": 720, "upscale_max_resolution": 1280})
    project["shots"] = [{"id": "temp_shot", "name": "验证镜头", "mode": "t2v",
                         "prompt": "A red apple on a wooden table, slow camera push in.",
                         "duration": 2, "seed": 5, "selected": True}]
    store.change(lambda data: data["projects"].__setitem__(project_id, project))

    run_id = store.uid()
    run = {"id": run_id, "project_id": project_id, "shot_ids": ["temp_shot"],
           "status": "queued", "current": "", "results": [], "created_at": time.time()}
    store.change(lambda data: data["runs"].__setitem__(run_id, run))
    threading.Thread(target=runner.run_project, args=(run_id,), daemon=True).start()
    print(f"run {run_id} started for project {project_id}", flush=True)

    deadline = time.time() + 600
    reached = False
    while time.time() < deadline:
        status = store.read()["runs"][run_id]
        if status.get("status") == "awaiting_upscale":
            reached = True
            print("REACHED awaiting_upscale -> 现在把项目超分关掉", flush=True)
            break
        if status.get("status") in ("completed", "failed", "cancelled"):
            print("run ended before the confirmation:", status.get("status"),
                  status.get("current"), flush=True)
            break
        time.sleep(1)

    if reached:
        store.change(lambda data: data["projects"][project_id].update({"upscale": False}))
        print("项目超分已关闭，等待运行自动放行…", flush=True)
        released = None
        while time.time() < deadline:
            status = store.read()["runs"][run_id]
            if status.get("status") != "awaiting_upscale":
                released = time.time()
                break
            time.sleep(0.5)
        status = store.read()["runs"][run_id]
        print("RELEASED:", status.get("status"), "| skip_upscale=",
              status.get("skip_upscale"), "| reason=",
              status.get("upscale_skipped_reason"), flush=True)

        # let it finish
        while time.time() < deadline:
            status = store.read()["runs"][run_id]
            if status.get("status") in ("completed", "failed", "cancelled"):
                print("FINAL:", status.get("status"), "|", status.get("current"), flush=True)
                print("results:", json.dumps(status.get("results"), ensure_ascii=False)[:300])
                break
            time.sleep(2)

    # clean up the throwaway project and run
    store.change(lambda data: (data["projects"].pop(project_id, None),
                               data["runs"].pop(run_id, None)))
    print("临时项目已清理", flush=True)


if __name__ == "__main__":
    main()
