package edu.cwru.sepia.agent.planner.actions;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.environment.model.state.ResourceType;

public class DepositWood implements StripsAction {
	private GameState.Peasant peasant;
	private GameState.TownHall townHall;
	
	public DepositWood(GameState.Peasant peasant, GameState.TownHall townHall) {
		this.peasant = peasant;
		this.townHall = townHall;
	}
	
	public GameState.Peasant getPeasant() {
		return this.peasant;
	}
	
	public GameState.TownHall getTownHall() {
		return this.townHall;
	}
	
	public boolean preconditionsMet(GameState gameState) {
		GameState.Peasant peasant = gameState.getPeasants().get(this.peasant.getID());
		GameState.TownHall townHall = gameState.getTownHall();
		
		if (peasant == null || townHall.getID() != this.townHall.getID()) {
			return false;
		}
		
		return peasant.getCargoType() == this.peasant.getCargoType() &&
				peasant.getCargoType() == ResourceType.WOOD &&
				peasant.getCargoAmount() == this.peasant.getCargoAmount() &&
				peasant.getCargoAmount() > 0;
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		GameState.Peasant resultPeasant = result.getPeasants().get(this.peasant.getID());
		result.addWood(resultPeasant.getCargoAmount());
		resultPeasant.deposit();
		
		return result;
	}
	
	/**
	 * Assume the number of turns required to do a deposit is the number of moves to get from the
	 * peasant to the town hall, plus 1 for the actual deposit. 
	 */
	public double getCost() {
		return (this.peasant.getPosition().chebyshevDistance(this.townHall.getPosition())) + 1;
	}
}
