# Connecting Poetry Environment with Jupyter Notebook

This guide provides simple steps to connect a Poetry-managed environment to Jupyter Notebook.

## Steps

1. **Activate the Poetry Environment**

   First, navigate to your project directory and activate the Poetry shell:

   ```bash
   poetry shell
   ```

2. **Install Jupyter as a Development Dependency**

   If Jupyter is not already installed, add it as a development dependency:

   ```bash
   poetry add --group dev jupyter
   ```

3. **Register the Poetry Environment as a Jupyter Kernel**

   Run this command to make the Poetry environment available as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name=llmdataparser-env --display-name "Python (LLMDataParser)"
   ```

   - `--name=llmdataparser-env`: Assigns a name to the kernel.
   - `--display-name "Python (LLMDataParser)"`: Sets the display name seen in Jupyter.

4. **Start Jupyter Notebook**

   Launch Jupyter Notebook from the Poetry shell:

   ```bash
   jupyter notebook
   ```

5. **Select the Poetry Kernel in Jupyter**

   - Open a notebook in Jupyter.
   - Go to "Kernel" > "Change kernel" and select **Python (LLMDataParser)** from the list.

   This connects the notebook to your Poetry environment.

---

You’re now set up to use your Poetry environment within Jupyter Notebook!
