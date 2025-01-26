package research.neuralnetwork.loader.image;

import research.neuralnetwork.loader.AbstractMetaData;

public class ImageMetaData extends AbstractMetaData {
	private int height;
	private int width;
	
	public int getHeight() {
		return height;
	}
	
	public void setHeight(int height) {
		this.height = height;
	}
	
	public int getWidth() {
		return width;
	}
	
	public void setWidth(int width) {
		this.width = width;
	}

	@Override
	public void setItemsRead(int itemsRead) {
		super.setItemsRead(itemsRead);
		super.setTotalItemsRead(super.getTotalItemsRead()+itemsRead);
	}
	
	
}
