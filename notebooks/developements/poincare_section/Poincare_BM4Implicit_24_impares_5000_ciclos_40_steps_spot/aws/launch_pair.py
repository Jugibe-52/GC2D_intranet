"""Launch the two explicitly authorized independent Spot studies in Frankfurt.
Run inside the signed-in AWS CloudShell, using its normal account credentials.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import boto3
ROOT=Path(__file__).resolve().parent
s3=boto3.client('s3',region_name='eu-central-1')
ec2=boto3.client('ec2',region_name='eu-central-1')
configs=json.loads((ROOT/'experiments.json').read_text())
subnet='subnet-097898b8a21581c16';security_group='sg-0d625b968fe142b61';ami='ami-042dc8681de073ac4'
quota=boto3.client('service-quotas',region_name='eu-central-1').get_service_quota(ServiceCode='ec2',QuotaCode='L-34B43A08')['Quota']['Value']
assert quota>=32,quota
image=ec2.describe_images(ImageIds=[ami])['Images'][0]
assert image['Architecture']=='x86_64' and image['State']=='available'
network=ec2.describe_subnets(SubnetIds=[subnet])['Subnets'][0]
assert network['State']=='available'
sg=ec2.describe_security_groups(GroupIds=[security_group])['SecurityGroups'][0]
assert sg['VpcId']==network['VpcId'] and sg['IpPermissions']==[], 'Expected the existing no-inbound group'
for cfg in configs:
 folder=ROOT/cfg['folder'];m=json.loads((folder/'input_manifest.json').read_text())
 archive=folder/m['archive']
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['sha256']
 for file,key in [(archive,m['key']),(folder/'input_manifest.json','runs/'+cfg['run_id']+'/input_manifest.json'),(folder/'launch_user_data.sh','runs/'+cfg['run_id']+'/launch_user_data.sh')]:
  s3.upload_file(str(file),cfg['bucket'],key)
 print(json.dumps({'input_verified_and_uploaded':cfg['run_id'],'sha256':m['sha256']}),flush=True)
records=[]
for cfg in configs:
 folder=ROOT/cfg['folder'];user_data=(folder/'launch_user_data.sh').read_text()
 tags=[{'Key':'Name','Value':cfg['run_id']},{'Key':'StudyRunId','Value':cfg['run_id']},{'Key':'Purpose','Value':'Poincare study'},{'Key':'AutoStop','Value':'enabled'}]
 request=dict(ImageId=ami,InstanceType=cfg['instance_type'],MinCount=1,MaxCount=1,ClientToken=cfg['run_id'],
  IamInstanceProfile={'Name':'GC2DWorkerRole'},UserData=user_data,
  NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':subnet,'Groups':[security_group],'AssociatePublicIpAddress':True,'DeleteOnTermination':True}],
  BlockDeviceMappings=[{'DeviceName':image['RootDeviceName'],'Ebs':{'VolumeSize':32,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],
  InstanceMarketOptions={'MarketType':'spot','SpotOptions':{'MaxPrice':str(cfg['spot_max_price_usd_per_hour']),'SpotInstanceType':'persistent','InstanceInterruptionBehavior':'stop'}},
  InstanceInitiatedShutdownBehavior='stop',MetadataOptions={'HttpTokens':'required','HttpEndpoint':'enabled','HttpPutResponseHopLimit':1},
  TagSpecifications=[{'ResourceType':resource,'Tags':tags} for resource in ('instance','volume')])
 instance=ec2.run_instances(**request)['Instances'][0]
 record={**cfg,'instance_id':instance['InstanceId'],'spot_request_id':instance.get('SpotInstanceRequestId'),'availability_zone':instance['Placement']['AvailabilityZone'],
   'launch_time_utc':instance['LaunchTime'].isoformat(),'state':instance['State']['Name'],
   'input_sha256':json.loads((folder/'input_manifest.json').read_text())['sha256'],
   'user_data_sha256':hashlib.sha256(user_data.encode()).hexdigest(),
   'root_volume_gib':32,'root_volume_encrypted':True,'imds_v2_required':True,'security_group':security_group,'iam_instance_profile':'GC2DWorkerRole'}
 if record['spot_request_id']:ec2.create_tags(Resources=[record['spot_request_id']],Tags=tags)
 (folder/'launch_record.json').write_text(json.dumps(record,indent=2)+'\n')
 s3.upload_file(str(folder/'launch_record.json'),cfg['bucket'],'runs/'+cfg['run_id']+'/launch_record.json')
 records.append(record)
 print('LAUNCHED '+json.dumps(record),flush=True)
(ROOT/'launch_records.json').write_text(json.dumps(records,indent=2)+'\n')
