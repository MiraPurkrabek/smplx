import os

NUM_POSES = 2
NUM_VIEWS = 2


for view in [
    "FRONT",
    "BACK",
    "SIDE",
    "PERIMETER",
    "TOP",
    "BOTTOM",
]:
    cmd = "python examples/sample_random_poses.py --gt-type DEPTH --crop --num-poses {:d} --num-views {:d} --pose-simplicity 1.5 --view-preference {:s} --rotation 0".format(
        NUM_POSES, NUM_VIEWS,
        view
    )
    os.system(cmd)

for rotation in [
    45, 90, 180, 135, 180
]:
    cmd = "python examples/sample_random_poses.py --gt-type DEPTH --crop --num-poses {:d} --num-views {:d} --pose-simplicity 1.5 --view-preference FRONT --rotation {:d}".format(
        NUM_POSES, NUM_VIEWS,
        rotation
    )
    os.system(cmd)

for simplicity in [
    0.8, 1.0, 2.0
]:
    cmd = "python examples/sample_random_poses.py --gt-type DEPTH --crop --num-poses {:d} --num-views {:d} --pose-simplicity {:f} --view-preference FRONT --rotation 0".format(
        NUM_POSES, NUM_VIEWS,
        simplicity
    )
    os.system(cmd)
