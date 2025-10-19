
## Data Mining Midterm Project NS2284 Naga Sathwik Sangaraju
It implements Brute Force, Apriori, and FP-Growth algorithms to identify frequent itemsets and association rules from multiple datasets.
## Requirements
```
pip install -r requirements.txt
```
this will install requried librarys
```
pandas
mlxtend
```
## How to Run the Project

1. Open your terminal and go to the project root directory:
   ```
   cd path\cs634_midterm_Naga_Sathwik_sangaraju
   ```

2. Run the main script:
   ```
   python src/run_all.py
   ```

3. The program will:
   - Display available datasets (1–5)
   - Ask you to select one by number
   - Ask for minimum support and confidence
   - Run **Brute Force**, **Apriori**, and **FP-Growth** sequentially
   - Show all frequent itemsets and rules directly in the terminal
   - Save results inside the `outputs/` folder

## Running the Jupyter Notebook

To view everything in notebook form:

```
jupyter notebook notebooks/datamining.ipynb
```

Run all cells in order to display the outputs inline (in the secondcell we need to selct dataset and min support and confidence)

## Converting Notebook to Python

For converting the Jupyter notebook to Python script, use the following command:
```
jupyter nbconvert --to script notebooks/datamining.ipynb
```
This will create Python script named `datamining.py` in the same directory.

## Output Folder
All generated files are saved in the 'outputs/' folder, grouped by dataset name and then by the algorithm used.