import os


class SymbolicLinkCreator:
    def __init__(self, root_dir: str, target_suffix: str, num_links: int):
        """
        initiates the SymbolicLinkCreator object

        :param root_dir: dir to be scanned
        :param target_suffix: file suffix（e.g. ".ckpt"）
        :param num_links: numbers of symbolic links to be created
        """
        self.root_dir = root_dir
        self.target_suffix = target_suffix.lower()  # Ignore capitals and lower cases
        self.num_links = num_links

    def scan_and_create_links(self):
        """
        Recursively scans the directory and creates soft links for all files with the specified suffixes
        """
        for current_dir, _dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(self.target_suffix):
                    full_path = os.path.join(current_dir, file)
                    self.create_links_for_file(full_path)

    def create_links_for_file(self, file_path: str):
        """
        Creates a soft link to the specified file in the directory where it resides.

        :param file_path: abs path
        """
        dirname, filename = os.path.split(file_path)
        base, ext = os.path.splitext(filename)

        for i in range(1, self.num_links + 1):
            symlink_name = f"{base}_{i}{ext}"
            symlink_path = os.path.join(dirname, symlink_name)

            if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                os.remove(symlink_path)
            try:
                os.symlink(file_path, symlink_path)
                print(f"Created symlink: {symlink_path} -> {file_path}")
            except OSError as e:
                print(f"Error creating symlink for {file_path}: {e}")

    def delete_all_symlinks(self):
        """
        Recursively deletes all symbolic links under the specified root directory.
        Only soft links are removed, source files remain intact.
        """
        for current_dir, dirs, files in os.walk(self.root_dir):
            for file in files:
                full_path = os.path.join(current_dir, file)
                if os.path.islink(full_path):
                    os.remove(full_path)
                    print(f"Deleted symlink: {full_path}")

            for _dir in dirs:
                full_path = os.path.join(current_dir, _dir)
                if os.path.islink(full_path):
                    os.remove(full_path)
                    print(f"Deleted symlink directory: {full_path}")


# example
if __name__ == "__main__":
    model_base_path_list = ["/home/zhexu_umass_edu/PycharmProjects/SPRITE/model_compare/candidate_models"]

    toplevels_list = {}
    for path in model_base_path_list:
        candidate_model_path = path
        if candidate_model_path not in toplevels_list:
            symbolic_link_creator = SymbolicLinkCreator(
                root_dir=os.path.abspath(candidate_model_path), target_suffix=".ckpt", num_links=0
            )
            toplevels_list.update({candidate_model_path: symbolic_link_creator})
            symbolic_link_creator.scan_and_create_links()

    for _path, symbolic_link_creator in toplevels_list.items():
        symbolic_link_creator.delete_all_symlinks()
