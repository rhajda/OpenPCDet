import pathlib
from tools.comp_pipe.path_handle import paths
from tqdm import tqdm
import pickle

def create_if_needed(path: pathlib.Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
    return path

def copy_over(src_list: pathlib.Path, dst_folder: pathlib.Path, desc:str) -> None:
    [(dst_folder / src.name).write_bytes(src.read_bytes()) for src in tqdm(src_list, desc=desc) if not (dest / src.name).exists()]

def get_glob_regex(n:int, ending: str, factor:int = 1) -> str:
    if factor == 5:
        return f"*000[{0}-{2}][{0}-{9}][0-{9}]{ending}"
    return f"*000[{0}-{5}][0-{9}]{ending}"


def handle_pickle(src:str, target:str, n:int):
    objects = []
    with open(str(src), 'rb')  as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break

    create_if_needed(target)
    
    with open(str(target / src.name), 'wb') as f:
        for obj in tqdm(objects, desc=f"{src.name} to {target.name}"):
            try:
                for o in list(obj.keys()):
                    pickle.dump(obj[o][:n], f)
            except:
                pickle.dump(obj[:n], f)

if __name__ == "__main__":

    # Copy up to n data-samples in the same structure to the new location. 
    n = 50 
    tag = "real"
    path_to_local = paths.root / f"data_local/indy_up_to_{n}/{tag}"

    path_to_big_indy_dataset = paths.indy_real

    # handle "kitti_dbinfos_train.pkl"
    handle_pickle(path_to_big_indy_dataset / "kitti_dbinfos_train.pkl", path_to_local, n)

    # handle "kitti_infos_train.pkl"
    handle_pickle(path_to_big_indy_dataset / "kitti_infos_train.pkl", path_to_local, n)
    
    # handle "kitti_infos_val.pkl"
    handle_pickle(path_to_big_indy_dataset / "kitti_infos_val.pkl", path_to_local, n)
    

    # Copy "ImageSets"
    dest = create_if_needed(path_to_local / "ImageSets")
    content = (path_to_big_indy_dataset / "ImageSets").glob("*.txt")
    copy_over(content, dest, "ImageSets")

    # Copy "gt_database"
    dest = create_if_needed(path_to_local / "gt_database")
    src = path_to_big_indy_dataset / "gt_database"
    content = src.glob(get_glob_regex(n, "_Car_0.bin", 5))
    copy_over(content, dest, "gt_database")
     
    # Copy "bev"
    dest = create_if_needed(path_to_local / "training/bev")
    src = path_to_big_indy_dataset / "training/bev"
    content = src.glob(get_glob_regex(n, ".png", 1))
    copy_over(content, dest, "bev") 
    
    # Copy "velodyne"
    dest = create_if_needed(path_to_local / "training/velodyne")
    src = path_to_big_indy_dataset / "training/velodyne"
    # content = src.glob(get_glob_regex(n, ".pcd", 1))
    content = list(src.glob(f"*000[0-2][0-9][50].pcd")) + [src / "000300.pcd"]
    copy_over(content, dest, "velodyne") 
     
    # Copy "label_2d"
    dest = create_if_needed(path_to_local / "training/label_2")
    src = path_to_big_indy_dataset / "training/label_2"
    content = src.glob(get_glob_regex(n, ".txt", 1))
    copy_over(content, dest, "label_2")

    