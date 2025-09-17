# sdk-generator
The generator used to generate the external SDKs for msf

To install this, either: 
- run `pip install git+https://github.com/Mon-Systeme-Fourrager/sdk-generator.git` 
- add `git+https://github.com/Mon-Systeme-Fourrager/sdk-generator.git` to your requirements file

To publish a new version, update the version number in `pyproject.toml`, then run the following commands:
- `python -m build`   (Note: You might need to install build if not already done: `pip install build`)
- `twine check dist/*`    (Note: This is just to make sure the build has nothing missing. You might need to install as well: `pip install twine`)
- `git tag {version number} -m "{Release Message}"`
- `git push origin {version number}`
