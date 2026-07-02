import os
import glob

def baca_semua_sample():
    path_folder = os.path.join("data-sample", "text-base", "*.txt")
    files = glob.glob(path_folder)

    print(f"Menemukan {len(files)} file sample untuk diproses.\n")

    for file_path in files:
        nama_file = os.path.basename(file_path)
        print(f"====== Memproses: {nama_file} ======")

        with open(file_path, "r", encoding="utf-8") as f:
            isi_teks = f.read()
            # Di sini nanti Step 2 & 3 dari Workflow Agent lu bakal beraksi
            print(f"Isi file {nama_file} siap di-parse oleh Agent...")
            print("-" * 40)

if __name__ == "__main__":
    baca_semua_sample()
