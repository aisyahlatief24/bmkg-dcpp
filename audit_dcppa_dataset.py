
import os
import re
from collections import defaultdict

BASE_DIR = "data/raw"

variables = ["pr", "tas"]

for var in variables:
    folder = os.path.join(BASE_DIR, var)

    print(f"\n{'='*55}")
    print(f"AUDIT DCPP-A: {var.upper()}")
    print(f"{'='*55}")

    if not os.path.exists(folder):
        print("Folder tidak ditemukan:", folder)
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".nc") and "dcppA-hindcast" in f
    ]

    years = defaultdict(set)

    for filename in files:
        match = re.search(
            r"dcppA-hindcast_s(\d{4})-r(\d+)i",
            filename
        )

        if match:
            init_year = int(match.group(1))
            member = int(match.group(2))
            years[init_year].add(member)

    print("Jumlah file:", len(files))
    print("Jumlah tahun inisialisasi:", len(years))

    for year in sorted(years):
        members = sorted(years[year])
        print(
            f"{year}: {len(members)} member "
            f"(r{members[0]} sampai r{members[-1]})"
        )

    if years:
        print("Tahun paling awal:", min(years))
        print("Tahun paling akhir:", max(years))
