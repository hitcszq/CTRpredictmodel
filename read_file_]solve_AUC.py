import sklearn.metrics
pred_label_file = open("pred_labels.out")
try:
	index = 1
	for line in pred_label_file:
		line = line.rstrip()
		if "epoch" in line:
			print line
		elif "pos_bucket" in line:
			print line
		else:
			split_line = line.split(" ")
			if index%2 != 0:
				preds = split_line
				preds = map(eval,preds)
				print len(preds)
			else:
				labels = split_line
				labels = map(eval,labels)
				print len(labels)
				if sum(labels) == 0:
					print "this position labels are 0!not binary!"
				else:
					auc = sklearn.metrics.roc_auc_score(labels,preds)
					print("auc:%s"%(auc,))
			index += 1
finally:
	pred_label_file.close()
