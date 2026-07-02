import os
import subprocess

def check_venv_status() -> str:
    """Memeriksa apakah virtual environment (venv) sudah terinstall dengan benar di root proyek."""
    if os.path.exists("venv") and os.path.exists("venv/bin/activate"):
        return "✅ Virtual environment 'venv' ditemukan dan siap diaktifkan."
    return "❌ Virtual environment 'venv' tidak ditemukan di root directory, Mbaang!"

def generate_clean_requirements() -> str:
    """Mengotomatisasi pembuatan file requirements.txt yang bersih tanpa mengotori sistem."""
    essential_packages = ["fastapi", "uvicorn", "spacy", "pydantic", "python-multipart"]
    try:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            for package in essential_packages:
                f.write(f"{package}\n")
        return "✅ File requirements.txt berhasil dibuat dengan dependencies esensial!"
    except Exception as e:
        return f"❌ Gagal membuat requirements.txt: {str(e)}"

def verify_git_protection() -> str:
    """Memastikan file .gitignore sudah mengunci venv dan output_model demi keamanan cloud."""
    if not os.path.exists(".gitignore"):
        return "⚠️ File .gitignore belum dibuat, Mbaang!"

    with open(".gitignore", "r", encoding="utf-8") as f:
        content = f.read()

    warnings = []
    if "venv/" not in content:
        warnings.append("venv/")
    if "output_model/" not in content:
        warnings.append("output_model/")

    if warnings:
        return f"⚠️ Peringatan: File berikut belum dikunci di .gitignore: {', '.join(warnings)}"
    return "✅ Aman! .gitignore sudah mengunci semua file raksasa dengan benar."
