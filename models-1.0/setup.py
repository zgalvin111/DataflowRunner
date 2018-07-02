from setuptools import setup

setup(name='models',
	version='1.0',
	description='The greatest model ever created for toxic post classification',
	packages=['models'],
	include_package_data=True,
	package_data={'models':['my_model.h5','word_index_dic.pickle']},
	install_requres=['pathlib','keras'],
)
