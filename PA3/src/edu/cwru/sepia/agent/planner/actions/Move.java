package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class Move implements StripsAction {
	
	private Position source;
	private Position destination;
	
	public Move(Position source, Position destination) {
		this.source = source;
		this.destination = destination;
	}

	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasantView = state.getPeasantView();

		if (peasantView == null) {
			return false;
		}

		Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
		
		return peasantPosition.equals(source) &&
				!state.getStateView().isUnitAt(destination.x, destination.y) &&
				!state.getStateView().isResourceAt(destination.x, destination.y);
	}
	
	public GameState apply(GameState gameState) {
		State state;
		try {
			state = gameState.getStateView().getStateCreator().createState();
		} catch (IOException e) {
			return null;
		}
		
		Unit peasant = state.getUnit(gameState.getStateView().unitAt(source.x, source.y));
		
		state.moveUnit(peasant, source.getDirection(destination));
		
		return new GameState(state.getView(gameState.getPlayerNum()),
							gameState.getPlayerNum(),
							gameState.getRequiredGold(),
							gameState.getRequiredWood(),
							gameState.getBuildPeasants());
	}

}