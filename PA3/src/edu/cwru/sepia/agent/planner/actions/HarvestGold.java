package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;
import java.util.List;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class HarvestGold implements StripsAction{

	private GameState.Peasant peasant;
	private GameState.Resource mine;
	
	public HarvestGold(GameState.Peasant peasant, GameState.Resource mine){
		this.peasant = peasant;
		this.mine = mine;
	}
	
	public Position getPeasantPos(){
		return peasant.getPosition();
	}
	
	public Position getMinePos(){
		return mine.getPosition();
	}
	
	public boolean preconditionsMet(GameState gameState) {
		GameState.Peasant peasant = gameState.getPeasant();

		if (peasant == null) {
			return false;
		}
		
		//PRECONDITIONS:
		//There's an empty adjacent position next to the mine
		//The resource is a mine and not a forest
		//The mine has at least 100 gold remaining
		//The peasant isn't carrying any cargo
			
		return mine.getType() == ResourceNode.Type.GOLD_MINE &&
				mine.getAmountRemaining() >= 100 &&
				peasant.getCargoAmount() == 0;		
	}

	
	//TODO: We need to talk about the different between result and resultPeasant
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
	
	

}
