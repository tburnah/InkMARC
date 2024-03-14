using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Azure.Storage.Blobs;

namespace Scryv.Utilities
{
    public class AzureStorageService
    {
        private readonly string connectionString = "DefaultEndpointsProtocol=https;AccountName=scryv951e;AccountKey=xYKPPl0/mWA2xAn6hKSK/a3NZzaJ5WWZ6q846ciRKv1WfhjQ9OlS1jdDKeQ+n01L5v8VCggHNbpR+AStoEPlXg==;EndpointSuffix=core.windows.net";

        public async Task UploadFileAsync(string containerName, string blobName, Stream fileStream)
        {
            var blobServiceClient = new BlobServiceClient(connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobClient = containerClient.GetBlobClient(blobName);

            await blobClient.UploadAsync(fileStream, true);
        }

        public async Task DownloadFileAsync(string containerName, string blobName, string filePath)
        {
            var blobServiceClient = new BlobServiceClient(connectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient(containerName);
            var blobClient = containerClient.GetBlobClient(blobName);

            await blobClient.DownloadToAsync(filePath);
        }
    }
}
