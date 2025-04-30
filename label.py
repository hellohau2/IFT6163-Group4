
"""
label_grasp_frames.py
Press a key for each displayed frame:

    0 = fail            (cube untouched / dropped)
    1 = partial_far     (arm moving, cube >10 cm away)
    2 = partial_close   (gripper almost touching cube)
    3 = success         (cube inside gripper or lifted)
    d = delete image
    q = quit

Labeled images are MOVED into ./<label>/ .
"""
import cv2, glob, os, shutil, argparse

LABEL_MAP = {ord("0"): "fail",
             ord("1"): "partial_far",
             ord("2"): "partial_close",
             ord("3"): "success",
             ord("d"): "DELETE",
             ord("q"): "QUIT"}

def main(src):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(src, ext))
    files.sort()
    if not files:
        print("No images in", src)
        return

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("label", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    for f in files:
        img = cv2.imread(f)
        cv2.imshow("label", img)
        key = cv2.waitKey(0)
        if key not in LABEL_MAP:
            print("unknown key, skipping")
            continue
        action = LABEL_MAP[key]
        if action == "QUIT":
            break
        if action == "DELETE":
            os.remove(f); print("deleted", os.path.basename(f)); continue
        dst = os.path.join(src, action)
        os.makedirs(dst, exist_ok=True)
        shutil.move(f, os.path.join(dst, os.path.basename(f)))
        print(os.path.basename(f), "→", action)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="./reward_debug_images")
    main(ap.parse_args().folder)
