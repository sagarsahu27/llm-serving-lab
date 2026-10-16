// Single Azure GPU VM for the LLM serving lab.
// SSH is the only inbound port; reach the gateway/backends through an SSH tunnel.
// Deploy with infra/deploy.sh (or az deployment group create, see docs/runbook.md).

targetScope = 'resourceGroup'

@description('Prefix for resource names.')
@minLength(3)
@maxLength(16)
param namePrefix string = 'llmlab'

@description('Azure region. Must have quota for the chosen GPU family.')
param location string = resourceGroup().location

@description('GPU VM size. NC4as_T4_v3 = 1x T4 16 GB (cheapest); NC24ads_A100_v4 = 1x A100 80 GB.')
@allowed([
  'Standard_NC4as_T4_v3'
  'Standard_NC8as_T4_v3'
  'Standard_NC16as_T4_v3'
  'Standard_NC24ads_A100_v4'
])
param vmSize string = 'Standard_NC4as_T4_v3'

@description('Linux admin user name.')
param adminUsername string = 'azureuser'

@description('SSH public key (contents of ~/.ssh/id_ed25519.pub or id_rsa.pub).')
param sshPublicKey string

@description('Source CIDR allowed to SSH, e.g. "203.0.113.10/32". Avoid "*".')
param allowedSshSourcePrefix string

@description('OS disk size in GB (engine images are 10-20 GB each, plus model caches).')
@minValue(128)
param osDiskSizeGB int = 256

@description('Optional git URL cloned into ~/llm-serving-lab on first boot.')
param repoUrl string = ''

@description('Daily auto-shutdown time (HHmm) in autoShutdownTimeZone. Empty string disables it.')
param autoShutdownTime string = '1900'

@description('Windows time zone id for auto-shutdown, e.g. "UTC" or "India Standard Time".')
param autoShutdownTimeZone string = 'UTC'

@description('Use Spot pricing (much cheaper, can be evicted).')
param useSpot bool = false

var cloudInit = replace(
  replace(
    replace(
      loadTextContent('cloud-init.yaml'),
      '__BOOTSTRAP_B64__',
      loadFileAsBase64('../scripts/bootstrap_vm.sh')
    ),
    '__ADMIN_USER__',
    adminUsername
  ),
  '__REPO_URL__',
  repoUrl
)
var dnsLabel = toLower('${namePrefix}-${uniqueString(resourceGroup().id)}')

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: '${namePrefix}-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowSshFromAllowedSource'
        properties: {
          priority: 1000
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: allowedSshSourcePrefix
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: '${namePrefix}-vnet'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: ['10.42.0.0/16']
    }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: '10.42.0.0/24'
          networkSecurityGroup: {
            id: nsg.id
          }
        }
      }
    ]
  }
}

resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: '${namePrefix}-pip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: {
      domainNameLabel: dnsLabel
    }
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-11-01' = {
  name: '${namePrefix}-nic'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          privateIPAllocationMethod: 'Dynamic'
          subnet: {
            id: vnet.properties.subnets[0].id
          }
          publicIPAddress: {
            id: publicIp.id
          }
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-03-01' = {
  name: '${namePrefix}-vm'
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    priority: useSpot ? 'Spot' : 'Regular'
    evictionPolicy: useSpot ? 'Deallocate' : null
    billingProfile: useSpot ? { maxPrice: -1 } : null
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: osDiskSizeGB
        deleteOption: 'Delete'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
      }
    }
    osProfile: {
      computerName: '${namePrefix}-vm'
      adminUsername: adminUsername
      customData: base64(cloudInit)
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
          properties: {
            deleteOption: 'Delete'
          }
        }
      ]
    }
  }
}

resource autoShutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = if (!empty(autoShutdownTime)) {
  name: 'shutdown-computevm-${vm.name}'
  location: location
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: {
      time: autoShutdownTime
    }
    timeZoneId: autoShutdownTimeZone
    targetResourceId: vm.id
    notificationSettings: {
      status: 'Disabled'
    }
  }
}

output vmName string = vm.name
output publicIp string = publicIp.properties.ipAddress
output fqdn string = publicIp.properties.dnsSettings.fqdn
output sshCommand string = 'ssh ${adminUsername}@${publicIp.properties.dnsSettings.fqdn}'
output tunnelCommand string = './scripts/tunnel.sh ${adminUsername}@${publicIp.properties.dnsSettings.fqdn}'
