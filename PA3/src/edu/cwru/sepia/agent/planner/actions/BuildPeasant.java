package edu.cwru.sepia.agent.planner.actions;

import edu.cwru.sepia.agent.planner.GameState;

public class BuildPeasant implements StripsAction {
	final int PEASANT_GOLD_COST = 400;

	public boolean preconditionsMet(GameState gameState) {
		/* PRECONDITIONS:
		 * 	sufficient gold
		 *  sufficient food
		 */
		return gameState.getGoldTotal() >= PEASANT_GOLD_COST &&
				gameState.getPeasants().size() < 3;
	}

	@Override
	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		result.addPeasant();
		result.addGold(-PEASANT_GOLD_COST);
		return result;
	}

	@Override
	/**
	 * Building a peasant takes a single time step, so it has a cost of 1.
	 */
	public double getCost() {
		return 1;
	}
}
