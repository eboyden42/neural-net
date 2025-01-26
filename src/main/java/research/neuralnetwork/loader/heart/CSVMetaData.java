package research.neuralnetwork.loader.heart;

import research.neuralnetwork.loader.AbstractMetaData;

public class CSVMetaData extends AbstractMetaData {
	@Override
	public void setItemsRead(int itemsRead) {
		super.setItemsRead(itemsRead);
		super.setTotalItemsRead(super.getTotalItemsRead()+itemsRead);
	}
	
}
