package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class DepositGold implements StripsAction {

	private Position peasantPos;
	private Position townHallPos;
	
	public DepositGold(Position peasantPos, Position townHallPos){
		this.peasantPos = peasantPos;
		this.townHallPos = townHallPos;		
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasant = state.getUnits().get(0);
		//TODO: Check that peasant is next to town hall
		
		return peasant.getCargoAmount() == 0 && 
				peasant.getCargoType() == ResourceType.GOLD;
	}

	public GameState apply(GameState gameState) {
		State state;
		try {
			state = gameState.getStateView().getStateCreator().createState();
		} catch (IOException e) {
			return null;
		}
		
		Unit peasant = state.getUnit(gameState.getStateView().unitAt(peasantPos.x, peasantPos.y));
		
		peasant.setCargo(ResourceType.GOLD, 0);
		state.addResourceAmount(gameState.getPlayerNum(), ResourceType.GOLD, 100);
		
		return new GameState(state.getView(gameState.getPlayerNum()),
				gameState.getPlayerNum(),
				gameState.getRequiredGold(),
				gameState.getRequiredWood(),
				gameState.getBuildPeasants());	
	}
	
}
