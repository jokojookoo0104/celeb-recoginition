$(function() {
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);

        $.ajax({
               url : 'http://celeb.kyanon.digital/api/uploader',
               type : 'POST', //example post request here, made this for file
                              // transfer over flask
               headers: {
                 // VERY IMPORTANT, although for security purposes as i've read
                 // before, using a wildcard is unsafe, basically any other
                 // website can mimic my requests which isn't i am concerned
                 // about for most of my learning projects
                    'Access-Control-Allow-Origin': '*',
                  },
               data : form_data,
               processData: false,  // tell jQuery not to process the data
               contentType: false,  // tell jQuery not to set contentType
               success : function(data) {
                   console.log(data);
                   // alert(data);
                   document.write(data)
               }
        });
    });
});


