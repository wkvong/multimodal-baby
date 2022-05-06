from multimodal.object_categories_data_module import ObjectCategoriesDataModule

if __name__ == "__main__":
    object_categories_dm = ObjectCategoriesDataModule()
    object_categories_dm.prepare_data()
    object_categories_dm.setup()

    dataloader = object_categories_dm.test_dataloader()
    batch = next(iter(dataloader))
    print(batch)
