# Rationale
To model expected infield throw performance and develop an evaluation tool by developing a machine learning model using [PyCaret](https://pycaret.org/)

# Dataset Description
The dataset contains data of defensive throws made to first base and the consequent game result following the throw. The dataset describes the thrower identity and their baseball position (denoted by position baseball number 1-9), the receiver identity and their baseball position (i.e., 3 for first baseman), the receiver distance from first base, characteristics of the throw (e.g., kinematics, whether it bounced or deflected), and the result of the throw (e.g., whether the batter was out or safe, number of runs scored on play, runner(s) on base, and number of outs in the inning).

**Feature Definition Clarification**:
- **Coordinate Data**: Origin is at the tip of home plate; positive x is towards the first base dugout; positive y is towards the outfield; positive z is perpendicular to the field (i.e., towards the sky)
- **Kinematic Data**: Distance is measured in ft and velocity is measured in ft/s
- **Exchange Time**: Time elapsed to transfer the ball from glove after fielding to throw release; `NA` values are admissible because plays where an exchange does not occur is possible (i.e., barehanded plays)
- **Throw Position & Velocity Data**: Represents the coordinate position on the field and velocity of the throw at the moment of throw release
- **Batter Position & Velocity Data**: Represents the coordinate position on the field and velocity of the batter at the moment of throw release
- **Receiver Position Coordinate Data**: Represents the coordinate position on the field of the first baseman at the moment of throw release
- **Bounce Position & Velocity Data**: Represents the coordinate position on the field and velocity of a bounced throw immediately after the first bounce; no bounce occurred if the value is `NA`
- **Throw Deflected By Receiver**: Indicates whether a throw was deflected (0: no deflection, thus clean catch, 1: throw was deflected)
- **Start State and End State**: Describes the runners on base and out state before/after a throw (e.g., _2_2: runner on second, 2 outs; ____1: no runners on, 1 out; 123_0: bases loaded, no outs)

# Coordinate Data Visualization
![Coordinate Data Visualization](/Visualizations/position_coordinate_visualization.png)

# Feature Engineering
**Features to Exclude**:
- `throw_id`: only serves to uniquely identify throws for data management and has cardinality throughout; thus, has no value for modeling purposes
- `receiver_position`: has zero cardinality because all instances have `3` as its value; thus, has no value for modeling purposes

**Engineered Features**:
- `field_thrower_same`: indicates whether the fielder and thrower are the same player (0: different player, 1: same player) as the throw difficulty can vary depending on the involvement of players
- `throw_bounced`: indicates whether a throw bounced (0: no bounce, 1: bounce occurred)
- `throw_deflected`: indicates whether a throw was deflected (0: no deflection, 1: deflected)
- `team_bounced_throw_rate`: portion of throws to first by a team that bounced; reflects the ability of the team to make a clean throw to first
- `team_deflected_throw_rate`: portion of throws to first by a team that was deflected; reflects the ability of the team to make a clean throw to first
- `thrower_bounced_throw_rate`: portion of throws to first by a player that bounced; reflects the ability of the player to make a clean throw to first
- `thrower_deflected_throw_rate`: portion of throws to first by a player that was deflected; reflects the ability of the player to make a clean throw to first
- `runner_on_1st`: indicates whether there is a runner on first at the start of the play (0: no runner on, 1: runner on)
- `runner_on_2nd`: indicates whether there is a runner on second at the start of the play (0: no runner on, 1: runner on)
- `runner_on_3rd`: indicates whether there is a runner on third at the start of the play (0: no runner on, 1: runner on)
- `number_of_outs`: indicates the number of outs at the start of the play

Runners on base and the number of outs may influence a throw difficulty.

- `batter_out`: indicates whether the batter was retired by the throw to first (0: batter was safe, 1: batter was thrown out); this feature is the target feature
- `team_out_rate`: portion of outs recorded on throws by a team; reflects the defensive quality of the team
- `thrower_out_rate`: portion of outs recorded on throws by a player; reflects the defensive quality of the player
- `throw_distance`: distance of throw as measured from the throw release coordinate position to the receiver coordinate position
- `batter_distance_from_1b`: distance of batter from first base at moment of throw release
- `throw_velo_resultant`: resultant magnitude of the throw velocity components; reflects the holistic velocity of the throw

# Model Development
[PyCaret](https://pycaret.org/) enabled multiple machine learning algorithms to be tested efficiently.  
Macro F1 score was the model evaluation metric of interest due to the class imbalance in the `batter_out` target feature. The class imbalance should be expected because outs occur more frequently than safe in baseball.

Model validation yielded a macro F1 score of 0.9833 and a near-ideal ROC curve.

![ROC Curve](/Visualizations/roc_curve.png)

The `batter_dist_from_1b` was the most important feature in the model in predicting whether a throw to first was going to retire the batter or not. This makes sense baseball-wise because a quicker throw would be required to throw out a batter that is closer to first, which increases the level of difficulty of the throw and may affect the quality of the throw.

Even though the model is developed to be a classifier, the target class probabilities of predictions might be better suited for infielder talent evaluation and developing a metric that conveys the value of an infielder's throwing ability.

# Player Evaluation Tool
While a classification model is informative, a more beneficial method for player evaluation is utilizing the class probabilities of the target feature to quantify an infielder's throwing value.
The probabilities allow infielders to be evaluated against expected outcomes generated by the model.

Using the model and the throw data that is passed in, the probability of whether a throw will result in the batter being thrown out at first the is calculated. The probability of a throw being successful and the in-game result are taken into account to assign a value for a throw an infielder makes. Throws with a higher probability of being an out are given less credit to infielders if an out is recorded, but are penalized more if they do not result in an out. More difficult throws (i.e., model suggests a very slim probability the batter will be throw out at first) that result in an out at first are given more credit.

Using the model outcomes, the evaluation formula is:

***number of outs recorded above expectation per throw = (number of actual outs recorded on throws – expected number of outs on throws) / number of throws***

Focusing on the `1` class probability of the target feature, which represents the probability of a throw to first recording an out, the ***expected number of outs on throws*** metric is created by summing all the out probabilities (i.e., values in the `predicition_score_1` column in the table above) for all the throws to first made by a given infielder. This metric is then subtracted from the number of outs actually recorded by the given infielder’s throws to first and the difference is divided by the number of throws made to assess how they performed compared to model expectations on a per throw basis. The difference is divided to make the metric into a rate that does not favor a greater number of throws.

Hypothetical player evaluations can be along the lines of:
-	Player A, who is a Gold Glove-caliber infielder, made 0.7 more outs per throw than expected
-	Player B, who was in the top 5 among infielders among errors committed, made -0.1 outs per throw compared to the expected value, meaning that Player B was below what they were expected to be

**Example Class Probabilities for Throws by a Player of Interest**:

| batter_out | Class 0 (Safe) Probability | Class 1 (Out) Probability |
| ----------- | ----------- | ----------- |
| 0 | 0.9977 | 0.0023 |
| 1 | 0.0005 | 0.9995 |
| 1 | 0.0188 | 0.9812 |

For the first throw in the table above, if the batter was thrown out, the infielder would have been credited with 0.9977 outs recorded above expectation per throw. Since they did not, they are credited with a -0.0023 outs recorded above expectation per throw, meaning they are performing below expectation.

For the three throws in the table above, the player would have a 0.0057 number of outs recorded above expected per throw, suggesting that the player is performing almost as expected:
- 2 outs recorded
- 1.983 out probability sum
- 3 throws made

**Outs recorded above expected per throw**: `(2 - 1.983) / 3 = 0.0057`
