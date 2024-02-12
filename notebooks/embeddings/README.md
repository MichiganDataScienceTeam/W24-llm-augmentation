# Virtual Environments
The notebooks have requirements (described in `requirements.txt`) such as the openai library, regex, and pandas. A Python best practice for downloading packages necessary for your code is using **virtual environments**.

## Why Use Virtual Environments?
Virtual environments allow you to manage dependencies for different projects separately. By isolating the project's dependencies, you ensure that each project has access to only the packages it needs, avoiding conflicts between package versions and making it easier to manage package versions across multiple projects.

## Creating and Activating Virtual Environments
There are multiple ways to use virtual environments in Python, including through the use of Anaconda or the built-in venv module. Here, we focus on the venv module, which is included in the Python Standard Library.

### Using venv
1. Creating a Virtual Environment

To create a virtual environment, run the following command in your terminal (for macOS/Linux) or command prompt (for Windows):
```
python3 -m venv venv
```
This command creates a directory named `venv` in your current directory, which will contain the Python executable files, and a copy of the pip library which can be used to install other packages.

2. Activating the Virtual Environment

The virtual environment must be activated before use. Activation scripts are located within the venv directory.

On macOS and Linux:

```
source venv/bin/activate
```
On Windows:

```
./venv/Scripts/activate
```
or
```
./venv/bin/activate
```
(the directory might differ on your system)
After activation, your shell prompt will change to show the name of the activated virtual environment, indicating that any Python or pip commands you run will operate within the virtual environment.

3. Installing Dependencies

Once the virtual environment is activated, install the project's dependencies by running:

```
pip install -r requirements.txt
```
This command reads the requirements.txt file and installs the specified packages into the virtual environment.

4. Deactivating the Virtual Environment

To exit the virtual environment and use your global Python environment again, simply run:

```
deactivate
```
This command deactivates the virtual environment, returning you to your system’s global Python environment.
