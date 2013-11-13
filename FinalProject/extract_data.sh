# Extract TRAIN Data
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/request_info_train.sql -o data/request_info_train.txt
# echo "Request Train Data Returned"
psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/sale_info_train.sql -o data/sale_info_train.txt
echo "Sale Train Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/btf_info_train.sql -o data/btf_info_train.txt
# echo "BTF Train Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/od_info_train.sql -o data/od_info_train.txt
# echo "OD Train Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/opp_info_train.sql -o data/opp_info_train.txt
# echo "BTF OPP Train Data Returned"


# #Extract TEST Data
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/request_info_test.sql -o data/request_info_test.txt
# echo "Request Test Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/sale_info_test.sql -o data/sale_info_test.txt
# echo "Sale Test Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/btf_info_test.sql -o data/btf_info_test.txt
# echo "BTF Test Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/od_info_test.sql -o data/od_info_test.txt
# echo "OD Test Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/opp_info_test.sql -o data/opp_info_test.txt
# echo "BTF OPP Test Data Returned"


#Seperate these cause they take forever
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/recent_request_history_test.sql -o data/recent_request_history_train.txt
# echo "Recent History Train Data Returned"
# psql -d greenzone -h greenzone-db.internal.atlassian.com -p 5432 -f sql_queries/recent_request_history_train.sql -o data/recent_request_history_test.txt
# echo "Recent History Test Data Returned"


#Extract Text - Can't do this right yet...
# psql -d helpspot -h helpspot.internal.atlassian.com -p 5432 -f sql_queries/request_text.sql -o data/request_text.txt
# # echo "Request Text Returned"