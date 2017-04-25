curl -v -H 'Accept: text/csv' 'http://localhost:5984/market/_design/futures/_list/settlement/tx?startkey=\["2009-01-01"\]' > ./setdays.csv
