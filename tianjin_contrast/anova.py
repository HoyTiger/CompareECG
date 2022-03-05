import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

file = '/data/0shared/shijia/对比范例/features_hr_label.csv'
data = pd.read_csv(file)

features_add_str = 'feature1'
for i in range(2, 1025):
    features_add_str += f' + feature{i}'

formula = f'hr~ {features_add_str}'
# formula = 'abs_hrc~ '
# formula = 'ecgc~ '

anova_results = anova_lm(ols(formula,data).fit())
print(anova_results)