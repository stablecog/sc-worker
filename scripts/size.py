import requests
from tabulate import tabulate


def get_release_info(package, version):
    table = []
    package_url = f"https://pypi.org/pypi/{package}/json"
    response = requests.get(package_url)
    data = response.json()
    releases = data.get("releases", {})
    if version:
        release_info = releases.get(version, [])
    else:
        latest_version = max(
            releases.keys(), key=lambda x: [int(y) for y in x.split(".")]
        )
        release_info = releases[latest_version]
    release = release_info[0]
    size_in_bytes = release.get("size", 0)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return package, version, size_in_mb


with open("requirements.txt") as f:
    total_in_mb = 0
    table = []
    for line in f:
        line = line.strip()
        try:
            p, v = line.split("==")
            package, version, size_in_mb = get_release_info(p, v)
            total_in_mb += size_in_mb
            table.append([package, version, round(size_in_mb, 2)])
        except ValueError:
            print(f"Skipping {line}")
    table = sorted(table, key=lambda x: x[2], reverse=True)
    print("\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(tabulate(table, headers=["Package", "Version", "Size (MB)"]))
    print("\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
    print(f"Total size: {total_in_mb:.2f} MB")
    print("\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
