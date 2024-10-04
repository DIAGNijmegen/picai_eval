import os
from pathlib import Path
from subprocess import check_call

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist

HERE = Path(__file__).parent


def gitcmd_update_submodules():
    """
    Check if the package is being deployed as a git repository. If so, recursively
    update all dependencies.

        Returns:
        True if the package is a git repository and the modules were updated. False otherwise.
    """
    if os.path.exists(os.path.join(HERE, '.git')):
        check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        return True

    return False


class gitcmd_develop(develop):
    """
    Specialized packaging class that runs git submodule update --init --recursive
    as part of the update/install procedure.
    """

    def run(self):
        gitcmd_update_submodules()
        develop.run(self)


class gitcmd_install(install):
    """
    Specialized packaging class that runs git submodule update --init --recursive
    as part of the update/install procedure.
    """

    def run(self):
        gitcmd_update_submodules()
        install.run(self)


class gitcmd_sdist(sdist):
    """
    Specialized packaging class that runs git submodule update --init --recursive
    as part of the update/install procedure;.
    """

    def run(self):
        gitcmd_update_submodules()
        sdist.run(self)


if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='1.4.8',  # also update version in metrics.py -> version
        author_email='Joeran.Bosma@radboudumc.nl',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/DIAGNijmegen/picai_eval',
        project_urls={
            "Bug Tracker": "https://github.com/DIAGNijmegen/picai_eval/issues"
        },
        license='Apache 2.0',
        packages=['picai_eval', 'picai_eval.stat_util'],
        cmdclass={
            'develop': gitcmd_develop,
            'install': gitcmd_install,
            'sdist': gitcmd_sdist,
        },
    )
