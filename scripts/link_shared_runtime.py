"""Link an existing shared offline runtime into a NEW Git checkout (no installs)."""
import argparse
from pathlib import Path


def link_runtime(existing, checkout):
    existing=Path(existing).resolve();checkout=Path(checkout).resolve()
    if existing==checkout:raise ValueError("Use a new Git checkout, separate from the offline directory")
    if not (existing/"scripts/lab.sh").is_file() or not (checkout/"scripts/lab.sh").is_file():
        raise ValueError("Both directories must contain the lab project")
    links=(".venv-linux",".venv-sam",".runtime","runs","assets/sam3.1")
    required=(".venv-linux/bin/python",".venv-sam/bin/python",
              ".runtime/egl-jammy/EGL_RUNTIME.json","assets/sam3.1/sam3.1_multiplex.pt",
              "assets/sam3.1/SAM_MANIFEST.json")
    for relative in required:
        if not (existing/relative).is_file():raise ValueError("Missing existing runtime file: "+relative)
    # Validate the entire plan before creating any links; never overwrite files.
    plan=[]
    for relative in links:
        source=existing/relative;target=checkout/relative
        if not source.is_dir():raise ValueError("Missing shared directory: "+str(source))
        if not target.parent.resolve().is_relative_to(checkout):raise ValueError("Link parent leaves checkout")
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=source.resolve():
                raise ValueError("Destination already exists and differs: "+str(target))
        else:plan.append((source,target))
    for source,target in plan:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(source,target_is_directory=True)
        print("LINK:",target,"->",source)
    print("Shared runtime ready. Keep the original offline directory in place.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--existing",type=Path,required=True)
    a=p.parse_args();link_runtime(a.existing,Path(__file__).resolve().parents[1])


if __name__=="__main__":main()
