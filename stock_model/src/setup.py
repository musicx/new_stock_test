try:
    from setuptools import setup
except:
    from distutils.core import setup


PACKAGES = ["quantax"]


setup(
    name='quantax',
    version='0.01',
    description='locally minimized quantaxis data read functionality',
    long_description='locally minimized quantaxis data read functionality',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    install_requires= ['numpy', 'pandas'],
    entry_points=None,
    keywords=['quant'],
    author='musicx',
    author_email='musicxing@gmail.com',
    url='com.gmail.musicxing',
    license='MIT',
    packages=PACKAGES,
    include_package_data=True,
    zip_safe=True
)