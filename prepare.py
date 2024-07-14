import os
import shutil


def prepare_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for phase in ['train', 'val']:
        input_phase_dir = os.path.join(input_dir, phase)
        output_phase_dir = os.path.join(output_dir, phase)

        if not os.path.exists(output_phase_dir):
            os.makedirs(output_phase_dir)

        for root, _, files in os.walk(input_phase_dir):
            for file in files:
                if file.endswith(".jpg"):
                    # Get the subdirectory (e.g., 'mel')
                    sub_dir = os.path.basename(root)
                    # Create new filename
                    new_filename = f"{sub_dir}_{file.replace('_', '-')}"
                    # Move file to new location with new name
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output_phase_dir, new_filename)
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")


if __name__ == "__main__":
    input_directory = "HAM10000"
    output_directory = "datasets"
    prepare_dataset(input_directory, output_directory)
