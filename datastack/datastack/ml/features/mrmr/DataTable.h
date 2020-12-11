#ifndef _DataTable_H_
#define _DataTable_H_

class DataTable
{
public:
    float* data;
    float** data2d;
    long nsample;
    long nvar;
	int corelen;
    int* classLabel;
    int* sampleNo;
    int b_zscore;
    int b_discetize;
	
    DataTable();
	~DataTable();
	int buildData2d();
	void destroyData2d();
	void zscore(long indExcludeColumn, int b_discretize);
	void discretize(double threshold, int b_discretize);
	void setColumns (char **cols, int len);
	void setCore (char **cols, int len);
	char **getColumns();
	char **getCoreColumns();
	int getCoreLen();
	int getIndex (char *col);
	void dumpTab ();

private:
	char	**colnames;
	char	**corecol;
};

#endif
