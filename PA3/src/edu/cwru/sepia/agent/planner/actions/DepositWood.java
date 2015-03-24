package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class DepositWood implements StripsAction {

	private Position peasantPos;
	private Position townHallPos;
	
	public DepositWood(Position peasantPos, Position townHallPos){
		this.peasantPos = peasantPos;
		this.townHallPos = townHallPos;		
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasantView = state.getPeasantView();
		Unit.UnitView townHallView = state.getTownHallView();

		if (peasantView == null || townHallView == null) {
			return false;
		}
		
		Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
		Position townHallPosition = new Position(townHallView.getXPosition(), townHallView.getYPosition());
		return peasantPos.equals(peasantPosition) &&
				townHallPos.equals(townHallPosition) &&
				peasantView.getCargoAmount() == 0 &&
				peasantPosition.isAdjacent(townHallPosition) &&
				peasantView.getCargoType() == ResourceType.WOOD;
	}

	public GameState apply(GameState gameState) {
		State state;
		try {
			state = gameState.getStateView().getStateCreator().createState();
		} catch (IOException e) {
			return null;
		}
		
		Unit peasant = state.getUnit(gameState.getStateView().unitAt(peasantPos.x, peasantPos.y));
		
		peasant.setCargo(ResourceType.WOOD, 0);
		state.addResourceAmount(gameState.getPlayerNum(), ResourceType.WOOD, 100);
		
		return new GameState(state.getView(gameState.getPlayerNum()),
				gameState.getPlayerNum(),
				gameState.getRequiredGold(),
				gameState.getRequiredWood(),
				gameState.getBuildPeasants(),
				this);	
	}
	
}
