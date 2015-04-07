package edu.cwru.sepia.agent.planner.actions;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;

public class HarvestGold implements StripsAction{

	private GameState.Peasant peasant;
	private GameState.Resource mine;
	
	public HarvestGold(GameState.Peasant peasant, GameState.Resource mine){
		this.peasant = peasant;
		this.mine = mine;
	}
	
	public GameState.Peasant getPeasant() {
		return this.peasant;
	}
	
	public GameState.Resource getMine() {
		return this.mine;
	}
	
	public boolean preconditionsMet(GameState gameState) {
		GameState.Peasant peasant = gameState.getPeasant();
		GameState.Resource mine = gameState.getResources().get(this.mine.getID());
		
		if (peasant.getID() != this.peasant.getID() || mine == null) {
			return false;
		}
		
		//PRECONDITIONS:
		//There's an empty adjacent position next to the mine
		//The resource is a mine and not a forest
		//The mine has at least 100 gold remaining
		//The peasant isn't carrying any cargo
			
		return mine.getType() == this.mine.getType() &&
				mine.getType() == ResourceNode.Type.GOLD_MINE &&
				mine.getAmountRemaining() == this.mine.getAmountRemaining() &&
				mine.getAmountRemaining() >= 100 &&
				peasant.getCargoAmount() == this.peasant.getCargoAmount() &&
				peasant.getCargoAmount() == 0;		
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		GameState.Peasant resultPeasant = result.getPeasant();
		
		//POSTCONDITIONS:
		//Reduce the amount in the mine
		//Increase the amount of Gold the peasant has.
		result.getResources().get(this.mine.getID()).harvest(100);
		resultPeasant.harvest(100, ResourceType.GOLD);;

		return result;
	}
	
	/**
	 * Assume the cost is the number of squares between the peasant and the mine, plus one more
	 * turn to do the actual harvesting.
	 */
	public double getCost() {
		return (this.peasant.getPosition().chebyshevDistance(this.mine.getPosition())) + 1;
	}
}