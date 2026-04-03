"""MCAP GT Generation - L20 Cluster Job Submission Script.

Packages code, uploads to cluster, and submits via aidi-inf-cli.

Usage:
    # Normal run (2 GPU)
    python submit.py --cluster carizon-4dgt-gpu --gpu 2

    # Sleep mode for debugging
    python submit.py --cluster carizon-4dgt-gpu --gpu 2 --sleep

    # With parking ID filter
    python submit.py --cluster carizon-4dgt-gpu --gpu 2 --parking-ids P1 P2
"""

import argparse
import os
import subprocess
import sys


def generate_job_yaml(cfg, upload_folder, run_cmd):
    yaml_name = "%s.yaml" % cfg["job_name"]
    with open(yaml_name, "w") as fn:
        fn.write("REQUIRED:\n")
        fn.write('  JOB_NAME: "%s"\n' % cfg["job_name"])
        fn.write('  JOB_PASSWD: "%s"\n' % cfg["job_password"])
        fn.write('  UPLOAD_DIR: "%s"\n' % os.path.basename(upload_folder))
        fn.write('  PROJECT_ID: "%s"\n' % cfg["project_id"])
        fn.write("  WORKER_MIN_NUM: %d\n" % cfg["num_machines"])
        fn.write("  WORKER_MAX_NUM: %d\n" % cfg["num_machines"])
        fn.write("  GPU_PER_WORKER: %d\n" % cfg["num_gpus_per_machine"])
        fn.write('  RUN_SCRIPTS: ' + run_cmd + "\n")
        fn.write("OPTIONAL:\n")
        fn.write("  PRIORITY: %s\n" % cfg["priority"])
        fn.write('  DOCKER_IMAGE: "%s"\n' % cfg["docker_image"])
        fn.write("  WALL_TIME: %d\n" % cfg["max_jobtime"])
        if cfg.get("input_bucket"):
            fn.write("  DATA_SPACE:\n")
            fn.write('    DATA_TYPE: "dmp"\n')
            fn.write('    INPUT: "%s"\n' % cfg["input_bucket"])
            if cfg.get("output_bucket"):
                fn.write('    OUTPUT: "%s"\n' % cfg["output_bucket"])
    return yaml_name


def generate_upload_folder(cfg, upload_folder):
    for path in cfg["folder_list"]:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        subprocess.check_call(["rsync", "-aL", path, upload_folder])
        print("Copied %s → %s" % (path, upload_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit MCAP GT job to L20 cluster")
    parser.add_argument("--cluster", type=str, default="carizon-4dgt-gpu")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--sleep", action="store_true", default=False)
    parser.add_argument("--parking-ids", nargs="+", default=None,
                        help="Parking lot IDs to pass through to orchestrator")
    parser.add_argument("--job-name", default=None)
    args = parser.parse_args()

    gpu_num = args.gpu
    max_jobtime = 59 if "debug" in args.cluster else 20160

    # Build run command
    parking_arg = ""
    if args.parking_ids:
        parking_arg = " --parking-ids " + " ".join(args.parking_ids)

    run_cmd_work = (
        "echo ${WORKING_PATH} && cd ${WORKING_PATH} "
        "&& bash ./run.sh " + ",".join(str(i) for i in range(gpu_num))
        + parking_arg
    )
    run_cmd_sleep = "cd ${WORKING_PATH} && bash ./run_sleep.sh"
    run_cmd = run_cmd_sleep if args.sleep else run_cmd_work

    k8s_config = dict(
        job_name=args.job_name or "mcap_gt_gen",
        job_password="1010",
        num_machines=1,
        num_gpus_per_machine=gpu_num,
        framework="pytorch",
        project_id="Carizon_general_project",
        # Buckets containing MCAP data
        input_bucket="carizon_collect_jfs4, carizon_maploc_jfs, carizon_fillback_jfs",
        output_bucket="carizon_perception_2",
        priority=5,
        docker_image=(
            "cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving.volcengine.com"
            "/imagesys/hat:fsd_multitask-cu11-20230616-v3.9"
        ),
        max_jobtime=max_jobtime,
        upload_folder_name="SuperGluePretrainedNetwork",
        folder_list=[
            "../SuperGluePretrainedNetwork",
            "./run.sh",
            "./run_sleep.sh",
        ],
        cluster=args.cluster,
    )

    print("=" * 50)
    print("MCAP GT Generation - Job Submission")
    print("=" * 50)
    for k, v in k8s_config.items():
        print(f"  {k}: {v}")
    print(f"  RUN_CMD: {run_cmd}")
    print("=" * 50)

    upload_folder = k8s_config["upload_folder_name"]
    subprocess.check_call(["mkdir", "-p", upload_folder])
    yaml_name = generate_job_yaml(k8s_config, upload_folder, run_cmd)
    assert os.path.exists(yaml_name), "Failed to generate YAML"

    generate_upload_folder(k8s_config, upload_folder)

    cmd = ["aidi-inf-cli", "job", "submit", "-f", yaml_name,
           "--current-cluster", k8s_config["cluster"]]
    print("Submitting:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
        print("Job submitted successfully.")
    except Exception as e:
        print(f"Submit failed: {e}")
        raise
    finally:
        if os.path.exists(upload_folder):
            subprocess.check_call(["rm", "-rf", upload_folder])
