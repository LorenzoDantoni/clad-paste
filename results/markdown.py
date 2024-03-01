import pandas as pd

df = pd.read_csv('csv/final_table.csv')

avgd = True

if avgd:

    df.drop(columns="Runtime", inplace=True)
    collapsed_df = df.sort_values(['Strategy','Memory Size'],ascending=True).groupby(["Model", "Strategy", "Memory Size"])

    collapsed_df = collapsed_df.mean()

    collapsed_df.drop(columns="Seed", inplace=True)

    collapsed_df.to_excel(f'md/collapsed_table.xlsx', index=True)
    collapsed_df.to_html(f'md/collapsed_table.html', index=True)

else:
    
    df = df.sort_values(['Strategy','Memory Size'],ascending=True).groupby(["Model", "Strategy", "Memory Size", "Seed"])
    df = df.first()
    df.to_excel(f'md/final_table.xlsx', index=True)
    df.to_html(f'md/final_table.html', index=True)
