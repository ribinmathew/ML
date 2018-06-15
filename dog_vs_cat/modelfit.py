
model.fit({'input':train_X},{'targets':train_Y},n_epoch=3,
          validation_set =({'input':test_x},{'target':test_y}),
          snapshot_step = 500,show_metric = True, run_id=Cat_vsDog)