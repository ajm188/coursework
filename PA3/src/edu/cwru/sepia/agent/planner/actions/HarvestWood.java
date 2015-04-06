package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;
import java.util.List;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.GameState.Resource;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class HarvestWood implements StripsAction{

	private GameState.Peasant peasant;
	private GameState.Resource forest;
	
	public HarvestWood(GameState state){
		this.peasant = state.getPeasant();
		this.forest = state.getForest();
	}
	
	public Position getPeasantPos(){
		return peasant.getPosition();		
	}
	
	public Position getForestPos(GameState state){
		 return this.forest.getPosition();
	}
	
	public boolean preconditionsMet(GameState state) {
		GameState.Peasant peasant = gameState.getPeasant();
		if (peasant == null) {
			return false;
		}
			
		return forest.getType() == ResourceNode.Type.TREE &&	
				forest.getAmountRemaining() >= 100 && 
				peasant.getCargoAmount() == 0;
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		GameState.Peasant resultPeasant = result.getPeasant();
		
		result.getResources().get(this.forest.getID()).harvest(100);
		resultPeasant.harvest(100,ResourceType.WOOD);
		
		return result;	
	}
}
