| Dataset             | Domain           | Feature            |   SHAP Before |   SHAP After | Bias Pattern                                                   |
|:--------------------|:-----------------|:-------------------|--------------:|-------------:|:---------------------------------------------------------------|
| UCI Adult           | Employment       | Occupation         |          0.31 |         0.12 | Favors male-dominated roles -> neutralized.                    |
| UCI Adult           | Employment       | Hours/Week         |          0.25 |         0.09 | Gendered work-hour imbalance -> reduced.                       |
| UCI Adult           | Employment       | Education          |          0.18 |         0.07 | Possible proxy bias -> stabilized across groups.               |
| COMPAS              | Criminal Justice | Prior Count        |          0.35 |         0.14 | Overestimates minority risk -> corrected via balancing.        |
| COMPAS              | Criminal Justice | Age                |          0.29 |         0.1  | Implicit bias via correlation -> controlled by stratification. |
| COMPAS              | Criminal Justice | Charge Degree      |          0.17 |         0.08 | Moderate case-type bias -> minor residual.                     |
| Diabetes 130-US     | Healthcare       | Insurance Type     |          0.33 |         0.11 | Socioeconomic access bias -> reduced via synthetic balancing.  |
| Diabetes 130-US     | Healthcare       | Medications        |          0.26 |         0.08 | Unequal treatment availability -> improved uniformity.         |
| Diabetes 130-US     | Healthcare       | Age                |          0.2  |         0.06 | Age-imbalance bias -> minor remaining influence.               |
| MEPS                | Healthcare       | Insurance Coverage |          0.28 |         0.1  | Coverage proxy for race -> balanced via CTGAN.                 |
| MEPS                | Healthcare       | Income Bracket     |          0.24 |         0.09 | Socioeconomic proxy -> attenuated.                             |
| MEPS                | Healthcare       | Region             |          0.16 |         0.07 | Regional access bias -> reduced.                               |
| Student Performance | Education        | Study Time         |          0.22 |         0.08 | Gendered study patterns -> normalized.                         |
| Student Performance | Education        | Failures           |          0.2  |         0.07 | Re-balanced through SMOTE.                                     |
| Student Performance | Education        | Family Support     |          0.18 |         0.06 | Reduced demographic-confounded influence.                      |
| Titanic             | Safety           | Sex                |          0.34 |         0.13 | Direct sensitive feature -> counterfactual mitigated.          |
| Titanic             | Safety           | Pclass             |          0.27 |         0.1  | Class-as-proxy -> reduced via balancing.                       |
| Titanic             | Safety           | Age                |          0.18 |         0.07 | Age proxy -> stabilized.                                       |
| CivilComments       | NLP              | Identity Tokens    |          0.36 |         0.13 | Identity terms drive toxicity score -> CTGAN paraphrase.       |
| CivilComments       | NLP              | Length             |          0.21 |         0.08 | Length-bias confound -> normalized.                            |
| CivilComments       | NLP              | Toxic n-grams      |          0.19 |         0.08 | Surface-cue reliance -> reduced.                               |