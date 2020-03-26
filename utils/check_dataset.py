import create_dataset as cdt

def check_dataset():

	data = cdt.create_data('None', True, 100)


	for i in range(10):
		print(str(i))
		print('----')
		print(data['states_train'][i]['Tgo'])
		print('----')
		print(data['states_train'][i]['stim1_dir'])
		print('----')
		print(data['states_train'][i]['task'])
		print('----')
		print('----')
			
	return
