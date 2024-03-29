package edu.cwru.sepia.agent.planner.actions;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;

public class HarvestWood implements StripsAction{

	private GameState.Peasant peasant;
	private GameState.Resource forest;
	
	public HarvestWood(GameState.Peasant peasant, GameState.Resource forest) {
		this.peasant = peasant;
		this.forest = forest;
	}
	
	public GameState.Peasant getPeasant() {
		return this.peasant;
	}
	
	public GameState.Resource getForest() {
		return this.forest;
	}
	
	public boolean preconditionsMet(GameState gameState) {
		GameState.Peasant peasant = gameState.getPeasants().get(this.peasant.getID());
		GameState.Resource forest = gameState.getResources().get(this.forest.getID());
		if (peasant == null || forest == null) {
			return false;
		}
			
		return forest.getType() == this.forest.getType() &&
				forest.getType() == ResourceNode.Type.TREE &&
				forest.getAmountRemaining() == this.forest.getAmountRemaining() &&
				forest.getAmountRemaining() >= 100 &&
				peasant.getCargoAmount() == this.peasant.getCargoAmount() &&
				peasant.getCargoAmount() == 0;
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		result.getResources().get(this.forest.getID()).harvest(100);
		result.getPeasants().get(this.peasant.getID()).harvest(100, ResourceType.WOOD);
		
		return result;	
	}
	
	/**
	 * Assume the cost is the number of squares between the peasant and the forest, plus one more
	 * turn to do the actual harvesting.
	 */
	public double getCost() {
		return (this.peasant.getPosition().chebyshevDistance(this.forest.getPosition())) + 1;
	}
}
