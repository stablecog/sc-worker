import requests


def get_release_info(package, version):
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
    print(f"{package} {version}: {size_in_mb:.2f} MB")
    return size_in_mb


with open("requirements.txt") as f:
    total_in_mb = 0
    for line in f:
        line = line.strip()
        try:
            package, version = line.split("==")
            mb = get_release_info(package, version)
            total_in_mb += mb
        except ValueError:
            print(f"Skipping {line}")
    print(f"Total: {total_in_mb:.2f} MB")
