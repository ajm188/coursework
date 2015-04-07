Andrew Mason (ajm188)
Steph Hippo (slh74)

GameState Heuristics:
100 * (goldToGo + woodToGo) / (Math.pow(this.peasants.size(), 3));

We give a state lower utility (preferred) if there are more peasants and fewer resources left to collect. This incentivizes the plan to create more peasants.

Parallelization for Multiple Peasants:
We elected not to write the Harvestk, etc actions. Instead, we had our PEAgent attempt to do some hand-wavy scheduling based on the sequential actions determined by the Planner. It converts the plan from a stack to a list in order to grab out actions that aren’t on the top of the stack. It then grabs the first action for each unique unit ID from the list and issues that action. It does not attempt to include the BuildPeasant action in the parallelization; that one is always done sequentially.

Running the Code:
Open the project in Eclipse and set up your run configurations to run it with whatever config file
you please. It does seem to crash sometimes on the midasLarge_BuildPeasant.xml, so beware of that.
