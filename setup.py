from setuptools import find_packages, setup

with open("deps/requirements.txt") as f:
    install_requires = f.read()


if __name__ == "__main__":
    setup(
        name="scale_rl",
        version="0.1.0",
        url="https://github.com/dojeon-ai/SimbaV2",
        license="Apache License 2.0",
        install_requires=install_requires,
        packages=find_packages(),
        python_requires=">=3.9.0",
        zip_safe=True,
    )
uv pip install jax==0.4.23 "jaxlib==0.4.23+cuda12.cudnn89" -f  https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Set uv cache to use tmpfs

# Or just use pip directly (slower but works)
export UV_CACHE_DIR=/tmp/uv_cache
pip install jax==0.4.23 "jaxlib==0.4.23+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --cache-dir /tmp/pip_cache


