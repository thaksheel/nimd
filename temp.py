from src import Interpretation 
import pandas as pd 

interpret = Interpretation() 

df_min = pd.read_excel("./exports/useful/noise_scores_results_minvol.xlsx")
df_min['model'] = ["minvol"] * len(df_min)
df_un = pd.read_excel("./exports/useful/noise_scores_results_un.xlsx")
df_un["model"] = ["unconstraint"] * len(df_un)
df = pd.concat([df_min, df_un])

interpret.plot_noise_score_influence_from_df(df)

print("END")
