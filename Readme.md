# Ideal Environment

## Constants
- **BUDGET**: 56.8
- **MIN_BID**: 0.5
- **MAX_BID**: 11.4
- **BID_UNIT**: 0.1
- **TEAM_SIZE**: 24
- **FORWARDS**: 14
- **DEFENSE**: 7
- **GOALIE**: 3
- **TEAMS**: 11

> **Note**: A smaller version of the environment is currently being used for testing.

## Player Pool Construction
The pool of available players is drafted from a CSV file named `players.csv`.

### Sample of `players.csv`
\```
"Player","Pos","Season","Team","GP","TOI","ATOI","Pts","W","SV","GA","FPV","FPV Adj","Pos Rank"
"Connor McDavid","F","22-23","EDM",81,1768,21.7,118,0,0,0,117,92,1
...
\```

## Auction Rules
1. Teams are randomly numbered.
2. Team 1 introduces a player from `players.csv` for auction, starting with a minimum bid of 0.5.
3. Subsequent teams can increase the bid.
4. A team can pass, forfeiting their chance to bid on the current player.
5. Bidding ends when only one team remains, who then wins the player.
6. The cycle repeats with the next team introducing a player.
7. The draft concludes when all teams have a full team.

## Team Composition Rules
- Budget constraints must be observed.
- Minimum budget must be reserved for unfilled team slots.

## Actions
- Introduce a player with a minimum bid of 0.5.
- Bid on a player, observing team and budget constraints.
- Pass on bidding.

## Reward Function
- The primary goal is to have the highest total points at the end of the draft.
- Some small, contextual rewards have been added.

## State Representation
- Includes environment constants.
- Information about the player up for auction.
- Information about all players in the auction.
- Information about the agent's team.
  
> **Note**: Information about other teams is currently excluded for simplicity.

## Model Architecture
- Actor-Critic model with epsilon initialization and decay.
- Adam optimizer.
- Kaiming weight initialization.

## Model Training
- Clipped logits to mitigate NaN issues.

## Debugging
- TensorBoard integration available, outputting to `runs/experiment_{current_time}`.

## Final Note
The ultimate goal is to pit this AI neural network against real humans in a draft. The AI should be capable of taking over a draft at any state and make informed decisions.
